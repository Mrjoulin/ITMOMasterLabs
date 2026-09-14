// frontend/tests/components/TaskModal.test.tsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import TaskModal from '../../components/TaskModal';
import { Task, Area, Priority } from '../../types';

const mockTask: Task = {
  id: '1',
  title: 'Test Task',
  description: 'Test content',
  area_id: '1',
  completed: false,
  due_date: new Date().toISOString(),
  priority: Priority.Medium,
};

const mockAreas: Area[] = [
  { id: '1', name: 'Work', color: 'bg-red-500' },
  { id: '2', name: 'Personal', color: 'bg-blue-500' },
];

describe('TaskModal Component', () => {
  const onClose = vi.fn();
  const onSave = vi.fn();

  beforeEach(() => {
    onClose.mockClear();
    onSave.mockClear();
  });

  it('renders correctly when open with a task', () => {
    render(
      <TaskModal
        isOpen={true}
        task={mockTask}
        onClose={onClose}
        onSave={onSave}
        areas={mockAreas}
      />
    );

    expect(screen.getByPlaceholderText('Task title')).toHaveValue(mockTask.title);
    expect(screen.getByPlaceholderText('Add a detailed description...')).toHaveValue(mockTask.description);
    expect(screen.getByDisplayValue('Work')).toBeInTheDocument();
    expect(screen.getByDisplayValue('Medium')).toBeInTheDocument();
  });

  it('does not render when closed', () => {
    const { container } = render(
        <TaskModal
          isOpen={false}
          task={mockTask}
          onClose={onClose}
          onSave={onSave}
          areas={mockAreas}
        />
      );
    expect(container.firstChild).toBeNull();
  });

  it('calls onClose when the close button is clicked', async () => {
    render(
        <TaskModal
          isOpen={true}
          task={mockTask}
          onClose={onClose}
          onSave={onSave}
          areas={mockAreas}
        />
      );

    await userEvent.click(screen.getByText('close'));
    expect(onClose).toHaveBeenCalled();
  });

  it('updates form data on input change', async () => {
    render(
        <TaskModal
          isOpen={true}
          task={mockTask}
          onClose={onClose}
          onSave={onSave}
          areas={mockAreas}
        />
      );

    const titleInput = screen.getByPlaceholderText('Task title');
    await userEvent.clear(titleInput);
    await userEvent.type(titleInput, 'New Title');
    expect(titleInput).toHaveValue('New Title');

    const descriptionTextarea = screen.getByPlaceholderText('Add a detailed description...');
    await userEvent.clear(descriptionTextarea);
    await userEvent.type(descriptionTextarea, 'New Content');
    expect(descriptionTextarea).toHaveValue('New Content');
  });

  it('calls onSave with updated data when Save is clicked', async () => {
    render(
        <TaskModal
          isOpen={true}
          task={mockTask}
          onClose={onClose}
          onSave={onSave}
          areas={mockAreas}
        />
      );

    const titleInput = screen.getByPlaceholderText('Task title');
    await userEvent.clear(titleInput);
    await userEvent.type(titleInput, 'Updated Title');
    await userEvent.click(screen.getByText('Save Changes'));

    expect(onSave).toHaveBeenCalledWith(expect.objectContaining({ title: 'Updated Title' }));
    expect(onClose).toHaveBeenCalled();
  });

  it('does not call onSave if title is empty', async () => {
    render(
        <TaskModal
          isOpen={true}
          task={mockTask}
          onClose={onClose}
          onSave={onSave}
          areas={mockAreas}
        />
      );

    const titleInput = screen.getByPlaceholderText('Task title');
    await userEvent.clear(titleInput);
    await userEvent.click(screen.getByText('Save Changes'));

    expect(onSave).not.toHaveBeenCalled();
    expect(onClose).not.toHaveBeenCalled();
  });

  it('toggles completion status', async () => {
    render(
        <TaskModal
          isOpen={true}
          task={mockTask}
          onClose={onClose}
          onSave={onSave}
          areas={mockAreas}
        />
      );
      
    const checkbox = screen.getByRole('button', { name: '' });
    await userEvent.click(checkbox);
    await userEvent.click(screen.getByText('Save Changes'));
    
    expect(onSave).toHaveBeenCalledWith(expect.objectContaining({ completed: true }));
  });
});
