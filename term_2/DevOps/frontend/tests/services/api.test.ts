// frontend/tests/services/api.test.ts
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { api } from '../../services/api';

// Mock global fetch
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('API Service', () => {
  beforeEach(() => {
    mockFetch.mockReset();
    localStorage.clear();
  });

  describe('Auth', () => {
    it('register stores token on success', async () => {
      const mockToken = { access_token: 'jwt-token' };
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockToken,
      });

      await api.register('test@ex.com', 'John', 'pass123');

      expect(localStorage.getItem('focusflow_token')).toBe('jwt-token');
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/users/register'),
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            email: 'test@ex.com',
            full_name: 'John',
            password: 'pass123',
          }),
        })
      );
    });

    it('register throws on duplicate email', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: async () => ({ detail: 'Email already exists' }),
      });

      await expect(api.register('dup@ex.com', 'Jane', 'pass')).rejects.toThrow(
        'Email already exists'
      );
      expect(localStorage.getItem('focusflow_token')).toBeNull();
    });

    it('login success stores token', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ access_token: 'token123' }),
      });

      const formData = new FormData();
      formData.append('username', 'test@ex.com');
      formData.append('password', 'pass');

      await api.login(formData);

      expect(localStorage.getItem('focusflow_token')).toBe('token123');
    });

    it('login failure does not store token', async () => {
        mockFetch.mockResolvedValueOnce({
            ok: false,
            json: async () => ({ detail: 'Incorrect username or password' }),
          });
    
          const formData = new FormData();
          formData.append('username', 'wrong@ex.com');
          formData.append('password', 'wrong');
    
          await expect(api.login(formData)).rejects.toThrow('Incorrect username or password');
    
          expect(localStorage.getItem('focusflow_token')).toBeNull();
    });

    it('clears token on 401 unauthorized', async () => {
      localStorage.setItem('focusflow_token', 'old-token');
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: async () => ({ detail: 'Unauthorized' }),
      });

      await expect(api.getUser()).rejects.toThrow('Unauthorized');
      expect(localStorage.getItem('focusflow_token')).toBeNull();
    });

    it('getUser fetches user data', async () => {
        const user = { email: 'test@test.com', full_name: 'Test' };
        mockFetch.mockResolvedValueOnce({
            ok: true,
            json: async () => user,
        });
        const result = await api.getUser();
        expect(result).toEqual(user);
    });
  });

  describe('Areas', () => {
    it('getAreas fetches areas', async () => {
        const areas = [{ id: 1, name: 'Area 1', color: '#fff' }];
        mockFetch.mockResolvedValueOnce({
            ok: true,
            json: async () => areas,
        });
        const result = await api.getAreas();
        expect(result).toEqual(areas);
    });

    it('createArea posts a new area', async () => {
        const newArea = { name: 'New Area', color: '#000' };
        mockFetch.mockResolvedValueOnce({
            ok: true,
            json: async () => newArea,
        });
        const result = await api.createArea('New Area', '#000');
        expect(result).toEqual(newArea);
    });
  });

  describe('Tasks', () => {
    it('getTasks fetches tasks', async () => {
        const tasks = [{ id: 1, title: 'Task 1' }];
        mockFetch.mockResolvedValueOnce({
            ok: true,
            json: async () => tasks,
        });
        const result = await api.getTasks();
        expect(result).toEqual(tasks);
    });

    it('createTask posts a new task', async () => {
        const newTask = { title: 'New Task' };
        mockFetch.mockResolvedValueOnce({
            ok: true,
            json: async () => newTask,
        });
        const result = await api.createTask(newTask);
        expect(result).toEqual(newTask);
    });

    it('updateTask patches an existing task', async () => {
        const updatedTask = { title: 'Updated Task' };
        mockFetch.mockResolvedValueOnce({
            ok: true,
            json: async () => updatedTask,
        });
        const result = await api.updateTask('1', updatedTask);
        expect(result).toEqual(updatedTask);
    });

    it('toggleTaskCompletion patches the completion status', async () => {
        const updatedTask = { completed: true };
        mockFetch.mockResolvedValueOnce({
            ok: true,
            json: async () => updatedTask,
        });
        const result = await api.toggleTaskCompletion('1', false);
        expect(result).toEqual(updatedTask);
    });
  });

  describe('Notes', () => {
    it('getNotes fetches notes', async () => {
        const notes = [{ id: 1, title: 'Note 1' }];
        mockFetch.mockResolvedValueOnce({
            ok: true,
            json: async () => notes,
        });
        const result = await api.getNotes();
        expect(result).toEqual(notes);
    });

    it('createNote posts a new note', async () => {
        const newNote = { title: 'New Note' };
        mockFetch.mockResolvedValueOnce({
            ok: true,
            json: async () => newNote,
        });
        const result = await api.createNote(newNote);
        expect(result).toEqual(newNote);
    });

    it('updateNote patches an existing note', async () => {
        const updatedNote = { title: 'Updated Note' };
        mockFetch.mockResolvedValueOnce({
            ok: true,
            json: async () => updatedNote,
        });
        const result = await api.updateNote('1', updatedNote);
        expect(result).toEqual(updatedNote);
    });
  });

  describe('Search', () => {
    it('search with query and type', async () => {
        const searchResults = [{ title: 'Result' }];
        mockFetch.mockResolvedValueOnce({
            ok: true,
            json: async () => searchResults,
        });
        const result = await api.search('test', 'task');
        expect(mockFetch).toHaveBeenCalledWith(expect.stringContaining('/search/?query=test&item_type=task'), expect.any(Object));
        expect(result).toEqual(searchResults);
    });

    it('search with only query', async () => {
        const searchResults = [{ title: 'Result' }];
        mockFetch.mockResolvedValueOnce({
            ok: true,
            json: async () => searchResults,
        });
        const result = await api.search('test', '');
        expect(mockFetch).toHaveBeenCalledWith(expect.stringContaining('/search/?query=test'), expect.any(Object));
        expect(result).toEqual(searchResults);
    });
  });
});
