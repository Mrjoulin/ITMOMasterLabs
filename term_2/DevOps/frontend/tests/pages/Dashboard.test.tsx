import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import Dashboard from '../../pages/Dashboard';
import { AppData, SearchResult, Priority } from '../../types';
import { api } from '../../services/api';

vi.mock('../../services/api');

const mockData: AppData = {
  user: { id: 1, full_name: 'Test User', email: 'test@example.com' },
  areas: [
    { id: 1, name: 'Work', color: 'bg-red-500' },
    { id: 2, name: 'Personal', color: 'bg-blue-500' },
  ],
  tasks: [
    { id: 1, title: 'Work Task 1', area_id: 1, completed: false, description: '', due_date: '', priority: Priority.High },
    { id: 2, title: 'Work Task 2', area_id: 1, completed: true, description: '', due_date: '', priority: Priority.High },
    { id: 3, title: 'Personal Task 1', area_id: 2, completed: false, description: '', due_date: '', priority: Priority.High },
  ],
  notes: [
    { id: 1, title: 'Work Note 1', area_id: 1, content: '' },
    { id: 2, title: 'Personal Note 1', area_id: 2, content: '' },
    { id: 3, title: 'Personal Note 2', area_id: 2, content: '' },
  ],
};

const waitForDebounce = () => new Promise(resolve => setTimeout(resolve, 350));

describe('Dashboard Page', () => {
  const onNavigate = vi.fn();
  const onNewTask = vi.fn();
  const onNewNote = vi.fn();
  const onOpenTask = vi.fn();
  const onOpenNote = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    vi.useRealTimers();
  });

  it('renders loading message when data is null', () => {
    render(<Dashboard data={null} onNavigate={onNavigate} onNewTask={onNewTask} onNewNote={onNewNote} onOpenTask={onOpenTask} onOpenNote={onOpenNote} />);
    expect(screen.getByText('Loading dashboard...')).toBeInTheDocument();
  });

  it('renders dashboard correctly with data', () => {
    const OriginalDate = global.Date;
    const mockDate = new OriginalDate(2026, 3, 17, 10);
    global.Date = class extends OriginalDate {
      constructor(...args: any[]) {
        if (args.length === 0) return mockDate;
        return new OriginalDate(...args);
      }
    } as DateConstructor;

    render(<Dashboard data={mockData} onNavigate={onNavigate} onNewTask={onNewTask} onNewNote={onNewNote} onOpenTask={onOpenTask} onOpenNote={onOpenNote} />);

    expect(screen.getByText(/Good morning/)).toBeInTheDocument();
    expect(screen.getByText(/Test/)).toBeInTheDocument();
    expect(screen.getByText('2')).toBeInTheDocument();
    expect(screen.getByText('3')).toBeInTheDocument();

    const mostActiveCard = screen.getByText('Most Active Area').closest('.rounded-xl');
    expect(mostActiveCard).toHaveTextContent('Personal');

    global.Date = OriginalDate;
  });

  it('calls onNewTask when "New Task" is clicked', async () => {
    render(<Dashboard data={mockData} onNavigate={onNavigate} onNewTask={onNewTask} onNewNote={onNewNote} onOpenTask={onOpenTask} onOpenNote={onOpenNote} />);
    await userEvent.click(screen.getByText('New Task'));
    expect(onNewTask).toHaveBeenCalledWith(null);
  });

  it('calls onNewNote when "New Note" is clicked', async () => {
    render(<Dashboard data={mockData} onNavigate={onNavigate} onNewTask={onNewTask} onNewNote={onNewNote} onOpenTask={onOpenTask} onOpenNote={onOpenNote} />);
    await userEvent.click(screen.getByText('New Note'));
    expect(onNewNote).toHaveBeenCalledWith(null);
  });

  it('triggers search and displays results', async () => {
    const searchResults: SearchResult[] = [{ id: 1, type: 'task', title: 'Work Task 1' }];
    (api.search as vi.Mock).mockResolvedValue(searchResults);

    render(<Dashboard data={mockData} onNavigate={onNavigate} onNewTask={onNewTask} onNewNote={onNewNote} onOpenTask={onOpenTask} onOpenNote={onOpenNote} />);

    const searchInput = screen.getByPlaceholderText('Search all...');
    await userEvent.type(searchInput, 'Work');
    await waitForDebounce();

    await waitFor(() => {
      expect(screen.getByText('Work Task 1')).toBeInTheDocument();
    });

    await userEvent.click(screen.getByText('Work Task 1'));
    expect(onOpenTask).toHaveBeenCalledWith(mockData.tasks[0]);
  });

  it('displays no results message', async () => {
    (api.search as vi.Mock).mockResolvedValue([]);

    render(<Dashboard data={mockData} onNavigate={onNavigate} onNewTask={onNewTask} onNewNote={onNewNote} onOpenTask={onOpenTask} onOpenNote={onOpenNote} />);

    const searchInput = screen.getByPlaceholderText('Search all...');
    await userEvent.type(searchInput, 'nonexistent');
    await waitForDebounce();

    await waitFor(() => {
      expect(screen.getByText('No results found.')).toBeInTheDocument();
    });
  });

  it('navigates when a stat card is clicked', async () => {
    render(<Dashboard data={mockData} onNavigate={onNavigate} onNewTask={onNewTask} onNewNote={onNewNote} onOpenTask={onOpenTask} onOpenNote={onOpenNote} />);

    await userEvent.click(screen.getByText('Active Tasks'));
    expect(onNavigate).toHaveBeenCalledWith('tasks');

    await userEvent.click(screen.getByText('Total Notes'));
    expect(onNavigate).toHaveBeenCalledWith('notes');

    await userEvent.click(screen.getByText('Most Active Area'));
    expect(onNavigate).toHaveBeenCalledWith('area:2');
  });
});